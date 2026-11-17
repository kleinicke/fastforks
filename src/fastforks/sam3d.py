"""Load Meta's SAM 3D estimators in their upstream environments with an original local snapshot."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    from sam_3d_body import SAM3DBodyEstimator  # ty: ignore[unresolved-import] - Meta's optional Body environment


class ObjectsPipeline(Protocol):
    """The callable interface of Meta's Objects notebook pipeline."""

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, seed: int | None = None, pointmap: object | None = None
    ) -> dict[str, Any]:
        """Reconstruct a masked object and return Meta's mesh, Gaussians and pose outputs."""
        ...


def load_body(
    snapshot: str | Path,
    *,
    device: str = "cuda",
    human_detector: object | None = None,
    human_segmentor: object | None = None,
    fov_estimator: object | None = None,
) -> SAM3DBodyEstimator:
    """Load the upstream Body estimator (ViT-H or DINOv3) and its MHR mesh model.

    Use the environment from Meta's sam-3d-body installation guide. The returned estimator exposes
    `process_one_image`, including its box, mask, camera-intrinsics and inference-type arguments.
    """
    root = Path(snapshot).expanduser().resolve()
    ckpt, mhr = root / "model.ckpt", root / "assets" / "mhr_model.pt"
    for path in (ckpt, mhr):
        if not path.is_file():
            raise FileNotFoundError(path)
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: PLC0415  # ty: ignore[unresolved-import] - Meta's optional Body environment

    model, config = load_sam_3d_body(str(ckpt), device=device, mhr_path=str(mhr))
    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=config,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )


def load_objects(pipeline: str | Path, *, checkout: str | Path, compile_model: bool = False) -> ObjectsPipeline:
    """Load the original Objects inference pipeline in Meta's sam-3d-objects environment.

    Supply the snapshot's pipeline.yaml and the installed upstream repository checkout. The resulting callable
    accepts an RGB image, a foreground mask, an optional seed and pointmap, and returns the upstream output dict.
    """
    config = Path(pipeline).expanduser().resolve()
    entry = Path(checkout).expanduser().resolve() / "notebook" / "inference.py"
    for path in (config, entry):
        if not path.is_file():
            raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("sam3d_objects_notebook_inference", entry)
    if spec is None or spec.loader is None:
        msg = f"Cannot load the SAM 3D Objects inference module at {entry}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Inference(str(config), compile=compile_model)
