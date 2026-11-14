"""Build the pinned Apache RF-DETR inference network from card architecture fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from fastforks._core.errors import MissingDependencyError, install_hint
from fastforks._core.patch import base_type, override

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Arch:
    """Architectural choices shared by RF-DETR tasks."""

    encoder: str
    out_feature_indexes: tuple[int, ...]
    vit_encoder_num_layers: int
    hidden_dim: int
    patch_size: int
    num_windows: int
    resolution: int
    positional_encoding_size: int
    dec_layers: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    dim_feedforward: int
    num_queries: int
    num_select: int
    group_detr: int
    projector_scale: tuple[str, ...]
    num_classes: int
    two_stage: bool
    bbox_reparam: bool
    lite_refpoint_refine: bool
    layer_norm: bool
    use_grouppose_keypoints: bool
    num_keypoints_per_class: tuple[int, ...]
    keypoint_cross_attn: bool
    inter_instance_kp_attn: bool
    grouppose_keypoint_dim_downscale: int
    dual_projector: bool
    dual_projector_kp_only: bool
    num_decoder_registers: int
    segmentation_head: bool
    mask_downsample_ratio: int
    trace_alpha: float


def network(a: Arch) -> torch.nn.Module:
    """Construct without downloading encoder weights; the caller loads its hash-checked checkpoint."""
    try:
        from .net.models._defaults import MODEL_DEFAULTS  # noqa: PLC0415 - optional libraries load lazily
        from .net.models.lwdetr import build_model  # noqa: PLC0415 - optional transformers / torchvision
    except ImportError as e:
        msg = f"RF-DETR needs transformers and torchvision: {install_hint('libs')} ({e})"
        raise MissingDependencyError(msg) from e
    args = {
        **asdict(MODEL_DEFAULTS),
        **asdict(a),
        "device": "cpu",
        "pretrain_weights": "hash-checked checkpoint",
        "force_no_pretrain": True,
        "freeze_encoder": False,
        "backbone_lora": False,
        "gradient_checkpointing": False,
        "drop_path": 0.0,
    }
    args = {k: list(v) if isinstance(v, tuple) else v for k, v in args.items()}
    return cast("torch.nn.Module", build_model(SimpleNamespace(**args)))


def export_path(net: Any) -> None:  # noqa: ANN401 - lazy vendored LWDETR API
    """Select the authors' inference path (`export()`: tuple outputs, precomputed position encodings), once."""
    if net._export:  # noqa: SLF001 - upstream flag
        return
    net.export()
    for module in net.modules():
        if "_forward_origin" in module.__dict__:
            # Upstream export saves its bound forward and installs a bound forward_export.
            # In inference no code restores that saved method; class overrides preserve eager destruction.
            override(module, base_type(module).forward_export)
            del module.__dict__["_forward_origin"]
