"""GMFlow optical-flow goldens from the reference implementation.

Reference: upstream autonomousvision/unimatch (`REF_COMMIT`), `unimatch/unimatch.py` `UniMatch` (`task="flow"`) with
the variant's checkpoint loaded strictly, CPU float32, the settings of `scripts/gmflow_evaluate.sh` for the variant
(`settings()`: scale1 defaults; scale2 `attn_splits_list 2 8`, `corr_radius_list -1 4`, `prop_radius_list -1 1`,
`upsample_factor 4`; regrefine6 `reg_refine` with `num_reg_refine 6`), the last entry of `flow_preds`.
Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp), passed
as upstream takes them (RGB in [0, 255]; upstream normalises; 384x512 needs no padding).
Stored as float32: `input` (image1, RGB), `input2` (image2, RGB) and `flow` ([2, H, W]: dx, dy in pixels).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/gmflow.py UNIMATCH_CHECKOUT [gmflow-scale1-things ...]

The checkout only has to contain `REF_COMMIT`; the reference runs from an export of that commit.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict

FAMILY = "gmflow"
UPSTREAM = "https://github.com/autonomousvision/unimatch"
REF_COMMIT = "943a99006950ab6c84ea72a72beb147fdd03c19a"


def settings(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Upstream constructor and call arguments of a variant (card `[params]`), as scripts/gmflow_evaluate.sh."""
    two = params["scales"] == 2  # noqa: PLR2004
    ctor = {"num_scales": params["scales"], "upsample_factor": 4 if two else 8, "reg_refine": params["refine"] > 0}
    call = {
        "attn_type": "swin",
        "attn_splits_list": [2, 8] if two else [2],
        "corr_radius_list": [-1, 4] if two else [-1],
        "prop_radius_list": [-1, 1] if two else [-1],
        "num_reg_refine": params["refine"] or 1,
        "task": "flow",
    }
    return ctor, call


def build(tree: Path, ckpt: Path, params: dict[str, Any]) -> torch.nn.Module:
    """Upstream `UniMatch` (from the exported `tree`, on `sys.path`) with the checkpoint, float32, eval."""
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    from unimatch.unimatch import UniMatch  # noqa: PLC0415  # reference checkout

    model = UniMatch(**settings(params)[0], task="flow")
    model.load_state_dict(load_state_dict(ckpt, key="model"), strict=True)
    return model.eval()


def calc_flow(
    model: torch.nn.Module, image1: torch.Tensor, image2: torch.Tensor, params: dict[str, Any]
) -> torch.Tensor:
    """Upstream's call on a `[B, 3, H, W]` pair in [0, 255] (H, W padded as the card says): the last flow."""
    return model(image1, image2, **settings(params)[1])["flow_preds"][-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of autonomousvision/unimatch")
    ap.add_argument("variants", nargs="*", help="variant ids (default: all)")
    a = ap.parse_args()
    import skimage  # noqa: PLC0415

    png = Path(skimage.data.__file__).parent / "astronaut.png"
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's astronaut.png"
        raise SystemExit(msg)
    y0, x0, h, w = CROP
    img1 = np.ascontiguousarray(skimage.io.imread(png)[y0 : y0 + h, x0 : x0 + w, :3])
    img2 = warp(img1)
    tree = reference_tree(a.checkout, REF_COMMIT)
    t1, t2 = (torch.from_numpy(i).permute(2, 0, 1)[None].float() for i in (img1, img2))
    for card, variant in registry.cards("flow").values():
        if card.family != FAMILY or (a.variants and variant.id not in a.variants):
            continue
        (ckpt,) = registry.weights(card, variant)
        params = dict(card.params | variant.params)
        model = build(tree, ckpt, params)
        with torch.inference_mode():
            flow = calc_flow(model, t1, t2, params)[0].numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "flow": flow.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream UniMatch {settings(params)}, skimage astronaut.png {h}x{w} crop "
            f"+ affine warp as raft_flow.py, torch {torch.__version__}",
            task="flow",
        )
        print(f"  {variant.id}: mean flow ({flow[0].mean():.3f}, {flow[1].mean():.3f}), |max| {np.abs(flow).max():.2f}")


if __name__ == "__main__":
    main()
