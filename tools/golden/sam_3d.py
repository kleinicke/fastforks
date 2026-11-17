"""Capture SAM 3D reference geometry using Meta's code and a local authorized snapshot.

Run in the corresponding upstream environment. Supply a single-person RGB image to Body, or an RGB
image and its binary foreground mask to Objects. The output NPZ preserves triangle indices, camera
translation and object pose separately. Upstream Body accepts both its ViT-H and DINOv3 snapshots.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BODY = "b5c765a0d89d789985e186d396315e7590887b94"
OBJECTS = "f91db411c50efee93d8db7aeb323885650f6f722"


def body(snapshot: Path, image: np.ndarray) -> dict[str, np.ndarray]:
    """One full-frame person using the original body, hand and MHR weights."""
    from fastforks.sam3d import load_body  # noqa: PLC0415 - run in Meta's environment

    estimator = load_body(snapshot)
    (person,) = estimator.process_one_image(image)
    return {"faces": estimator.faces, **{key: value for key, value in person.items() if value is not None}}


def objects(checkout: Path, snapshot: Path, image: np.ndarray, mask_path: Path, seed: int) -> dict[str, np.ndarray]:
    """One masked object with mesh, Gaussian centers and pose from the same seeded sample."""
    from fastforks.sam3d import load_objects  # noqa: PLC0415 - run in Meta's environment

    with Image.open(mask_path) as source:
        mask = np.asarray(source) > 0
    if mask.ndim == 3:  # noqa: PLR2004 - upstream uses the final channel for RGB / RGBA masks
        mask = mask[..., -1]
    if mask.shape != image.shape[:2] or not mask.any():
        msg = "Expected a nonempty foreground mask matching the input image dimensions"
        raise ValueError(msg)
    model = load_objects(snapshot / "checkpoints" / "pipeline.yaml", checkout=checkout)
    output = model(image, mask, seed=seed)
    mesh = output["glb"]
    return {
        "mask": mask,
        "vertices": np.asarray(mesh.vertices),
        "faces": np.asarray(mesh.faces),
        "gaussian_centers": output["gs"].get_xyz.detach().cpu().numpy(),
        **{key: output[key].detach().cpu().numpy() for key in ("rotation", "translation", "scale")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["body", "objects"])
    parser.add_argument("checkout", type=Path, help="Meta git checkout")
    parser.add_argument("snapshot", type=Path, help="original Hugging Face snapshot root")
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mask", type=Path, help="object foreground mask")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.model == "objects") != (args.mask is not None):
        parser.error("--mask is required for Objects and unused for Body")
    checkout = args.checkout.resolve()
    commit = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip()
    if commit != (BODY if args.model == "body" else OBJECTS):
        parser.error(f"Unexpected upstream commit {commit}")
    sys.path.insert(0, str(checkout))
    snapshot = args.snapshot.resolve()
    with Image.open(args.image) as source:
        image = np.array(source.convert("RGB"))
    if args.model == "objects":
        maps = objects(checkout, snapshot, image, args.mask, args.seed)
    else:
        maps = body(snapshot, image)
    if any(np.asarray(value).dtype.hasobject for value in maps.values()):
        msg = "Unexpected nonnumeric upstream output"
        raise ValueError(msg)
    with args.output.open("xb") as file:
        np.savez_compressed(file, image=image, upstream=commit, **maps)


if __name__ == "__main__":
    main()
