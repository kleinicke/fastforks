"""FoundationStereo goldens from the reference implementation.

Reference: upstream NVlabs/FoundationStereo (`REF_COMMIT`), CPU float32 (`mixed_precision=False`), 32 iterations,
upstream `InputPadder(divis_by=32)`. The reference needs timm, einops, omegaconf, huggingface_hub, torchvision and
scipy; run this with a Python that has them. Three shims (`_shim_*`) stand in for what upstream fetches from the
network or imports from its demo-only `Utils.py` (open3d, cv2, pandas); none changes the maths.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/foundationstereo.py FS_CHECKOUT WEIGHTS_DIR [vit-l ...]

`WEIGHTS_DIR` holds upstream's `pretrained_models/` layout (`<run>/model_best_bp2.pth` + `<run>/cfg.yaml`).
The checkout only has to contain `REF_COMMIT`; the reference runs from an export of that commit.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from _common import pair, reference_tree, write

if TYPE_CHECKING:
    from collections.abc import Callable

UPSTREAM = "https://github.com/NVlabs/FoundationStereo"
REF_COMMIT = "6e8806816b533e4d13ddbb95ffa907b797060a62"
VARIANTS = {"vit-l": "23-51-11", "vit-s": "11-33-40"}
ITERS = 32


def _shim_utils(root: Path) -> None:
    """Upstream's top-level `Utils.py` imports open3d / cv2 / pandas for the demo: keep only the two functions used."""
    tree = ast.parse((root / "Utils.py").read_text())
    keep = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in {"get_resize_keep_aspect_ratio", "freeze_model"}
    ]
    mod = types.ModuleType("Utils")
    code = compile(ast.Module(list[ast.stmt](keep), []), "Utils.py", "exec")
    exec(code, {"torch": torch, "np": np, "__name__": "Utils"}, mod.__dict__)  # noqa: S102 - our own parse of the reference
    sys.modules["Utils"] = mod


def _shim_timm() -> None:
    """No ImageNet download: the checkpoints hold the backbone weights."""
    import timm  # noqa: PLC0415 - reference-only dependency

    create = cast("Callable[..., torch.nn.Module]", timm.create_model)
    vars(timm)["create_model"] = lambda *a, **kw: create(*a, **{**kw, "pretrained": False})


def _shim_hub(root: Path) -> None:
    """`torch.hub.load('facebookresearch/dinov2', ...)` from the dinov2 copy in the checkout instead of GitHub."""
    load = torch.hub.load

    def local(repo: str, model: str, *a: object, **kw: object) -> torch.nn.Module:
        if repo != "facebookresearch/dinov2":
            raise ValueError(repo)
        return load(str(root / "dinov2"), model, *a, source="local", **kw)

    torch.hub.load = local  # reference shim


def main() -> None:
    from omegaconf import OmegaConf  # noqa: PLC0415  # reference-only dependency

    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    root = reference_tree(repo, REF_COMMIT)
    sys.path.insert(0, str(root))
    _shim_utils(root)
    _shim_timm()
    _shim_hub(root)
    from core.foundation_stereo import FoundationStereo  # noqa: PLC0415  # reference checkout on sys.path
    from core.utils.utils import InputPadder  # noqa: PLC0415  # reference checkout on sys.path

    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    padder = InputPadder(left.shape, divis_by=32)
    lp, rp = padder.pad(left, right)
    for name in sys.argv[3:] or VARIANTS:
        run = weights / VARIANTS[name]
        cfg = OmegaConf.load(run / "cfg.yaml")
        cfg.setdefault("vit_size", "vitl")  # as the upstream demo
        cfg["mixed_precision"] = False
        net = FoundationStereo(OmegaConf.create(cfg))
        net.load_state_dict(torch.load(run / "model_best_bp2.pth", map_location="cpu", weights_only=False)["model"])
        with torch.inference_mode():
            disp = net.eval()(lp, rp, iters=ITERS, test_mode=True)
        disp = padder.unpad(disp)[0, 0].numpy()
        write(
            f"foundationstereo-{name}@1",
            {"disparity": disp},
            family="foundationstereo",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, {ITERS} iters, timm/hub/Utils shims, torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
