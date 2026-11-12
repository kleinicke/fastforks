"""SAM 2.1 (image) goldens from the reference implementation.

Reference: upstream facebookresearch/sam2 (`REF_COMMIT`), CPU float32, `build_sam2` (default post-processing settings)
+ `SAM2ImagePredictor.set_image` on the synthetic image, then `predict` with the fixed prompts of `tests/synth.py` (a
centre point with multimask output; a box + the point, single mask). Stored: the low-resolution mask logits (clamped
to +-32, as returned) and IoU predictions (`box_` prefix for the second prompt).

Video (`video_*` keys): `build_sam2_video_predictor` + `SAM2VideoPredictor` on `tests/synth.py` `clip()` (frames
written as PNG and loaded with upstream `_load_img_as_tensor`, i.e. PIL bicubic resize), one object prompted on frame
0 per `clip_prompts()` entry (`add_new_points_or_box` / `add_new_mask`), `propagate_in_video` forward. Stored per
frame: the low-resolution mask logits after hole filling and the object score logits. Upstream's hole filling needs
its CUDA extension; `get_connected_components` is replaced by `scipy.ndimage.label` (8-connectivity, the same
labels / areas) so the CPU reference fills holes as the CUDA one does.

Run this with a Python that has the upstream dependencies (hydra-core, iopath, scipy).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/sam2.py SAM2_CHECKOUT WEIGHTS_DIR [tiny small base_plus large]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

from synth import clip, clip_prompts, prompts

UPSTREAM = "https://github.com/facebookresearch/sam2"
REF_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
VARIANTS = {  # size: (ref, config, checkpoint)
    "tiny": ("sam2.1-hiera-tiny", "t"),
    "small": ("sam2.1-hiera-small", "s"),
    "base_plus": ("sam2.1-hiera-base-plus", "b+"),
    "large": ("sam2.1-hiera-large", "l"),
}


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from sam2.build_sam import build_sam2  # noqa: PLC0415  # reference checkout
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415  # reference checkout

    _patch_video_loading()

    img = image()
    for key in sys.argv[3:] or VARIANTS:
        ref, cfg = VARIANTS[key]
        model = build_sam2(f"configs/sam2.1/sam2.1_hiera_{cfg}.yaml", str(weights / f"sam2.1_hiera_{key}.pt"), "cpu")
        pred = SAM2ImagePredictor(model)
        maps = {}
        with torch.inference_mode():
            pred.set_image(img)
            for prefix, p in prompts().items():
                _, iou, low = pred.predict(
                    point_coords=np.array(p["input_points"], np.float64),
                    point_labels=np.array(p["input_labels"]),
                    box=np.array(p["input_boxes"], np.float64) if "input_boxes" in p else None,
                    multimask_output=bool(p["multimask_output"]),
                )
                maps |= {f"{prefix}pred_masks": low, f"{prefix}iou_scores": iou}
        maps |= _video(cfg, weights / f"sam2.1_hiera_{key}.pt")
        write(
            f"{ref}@1",
            maps,
            family="sam2",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=(
                "CPU float32, build_sam2 + SAM2ImagePredictor set_image + predict (point; box + point); "
                "SAM2VideoPredictor on the synth clip (point; box; mask), scipy connected components, "
                f"torch {torch.__version__}"
            ),
            task="promptable_segmentation",
        )
        print(f"  iou {maps['iou_scores']}, box iou {maps['box_iou_scores']}")


def _patch_video_loading() -> None:
    """Upstream frame loading from a PNG folder (`_load_img_as_tensor`) and CPU connected components (scipy)."""
    import sam2.sam2_video_predictor as vp  # noqa: PLC0415  # reference checkout
    import scipy.ndimage  # noqa: PLC0415  # reference Python
    from sam2.utils import misc  # noqa: PLC0415  # reference checkout

    def load(video_path: str, image_size: int, offload_video_to_cpu: bool, **_: object) -> tuple:  # noqa: ARG001, FBT001
        paths = sorted(Path(video_path).glob("*.png"))
        mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
        loaded = [misc._load_img_as_tensor(str(p), image_size) for p in paths]  # noqa: SLF001
        images = torch.stack([x for x, _, _ in loaded])
        return (images - mean) / std, loaded[0][1], loaded[0][2]

    def components(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels, areas = torch.zeros_like(mask, dtype=torch.int32), torch.zeros_like(mask, dtype=torch.int32)
        for i in range(mask.shape[0]):
            lab, _ = scipy.ndimage.label(mask[i, 0].cpu().numpy(), structure=np.ones((3, 3)))
            count = np.bincount(lab.ravel())
            count[0] = 0
            labels[i, 0] = torch.from_numpy(lab.astype(np.int32))
            areas[i, 0] = torch.from_numpy(count[lab].astype(np.int32))
        return labels, areas

    vp.load_video_frames = load
    misc.get_connected_components = components


def _video(cfg: str, checkpoint: Path) -> dict[str, np.ndarray]:
    """Upstream `SAM2VideoPredictor` on the synth clip, one prompt kind at a time (`video*_` keys)."""
    import tempfile  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415
    from sam2.build_sam import build_sam2_video_predictor  # noqa: PLC0415  # reference checkout

    pred = build_sam2_video_predictor(f"configs/sam2.1/sam2.1_hiera_{cfg}.yaml", str(checkpoint), device="cpu")
    maps = {}
    with tempfile.TemporaryDirectory() as d, torch.inference_mode():
        for t, f in enumerate(clip()):
            Image.fromarray(f).save(Path(d) / f"{t:05d}.png")
        for prefix, p in clip_prompts().items():
            state = pred.init_state(d)
            if "input_masks" in p:
                pred.add_new_mask(state, 0, 1, p["input_masks"].astype(bool))
            else:
                pts = np.array(p["input_points"], np.float32) if "input_points" in p else None
                lab = np.array(p["input_labels"], np.int32) if "input_labels" in p else None
                box = np.array(p["input_boxes"], np.float32) if "input_boxes" in p else None
                pred.add_new_points_or_box(state, 0, 1, points=pts, labels=lab, box=box)
            for _ in pred.propagate_in_video(state):
                pass
            out = state["output_dict_per_obj"][0]
            frames = [out["cond_frame_outputs"][0]] + [out["non_cond_frame_outputs"][t] for t in range(1, len(clip()))]
            maps[f"{prefix}pred_masks"] = torch.cat([o["pred_masks"][:, 0] for o in frames]).numpy()
            maps[f"{prefix}object_score_logits"] = torch.cat([o["object_score_logits"][:, 0] for o in frames]).numpy()
            print(f"  {prefix}: object scores {np.round(maps[f'{prefix}object_score_logits'], 2)}")
    return maps


if __name__ == "__main__":
    main()
