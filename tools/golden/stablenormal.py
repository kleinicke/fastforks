"""StableNormal turbo goldens from the reference implementation.

Reference: upstream Stable-X/StableNormal (`REF_COMMIT`), `stablenormal/pipeline_yoso_normal.py` `YOSONormalsPipeline`
loaded as hubconf's `StableNormal_turbo` loads it (`from_pretrained` of the yoso-normal-v0-3 snapshot `HF_REVISION`
with `trust_remote_code`, `variant="fp16"`, `t_start=0`, no safety checker), CPU float32 (the fp16 files upcast), called
on the float image in [0, 1] with the defaults (`processing_resolution=768`, `match_input_resolution`, bilinear). The
Gaussian latent is passed as `latents`: a CPU `torch.Generator` seeded with `SEED` (float32), as the runners draw it
(upstream draws it from the unseeded global RNG). hubconf's `Predictor` wrapper (resize to 1024, background
segmentation) is not applied. The normals are stored in the camera frame (x right, y down, z forward): StableNormal's
(x left, y up, z to the viewer) -> (-x, -y, -z). Run this with a Python that has diffusers (`DIFFUSERS`) and
transformers; upstream pins diffusers 0.28 (its huggingface_hub API is gone), so the snapshot's `controlnetvae.py`
gets the moved `diffusers.models.controlnet` module under its old name.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/stablenormal.py STABLENORMAL_CHECKOUT YOSO_SNAPSHOT/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/Stable-X/StableNormal"
REF_COMMIT = "594b934630ab3bc71f35c77d14ec7feb98480cd0"
DIFFUSERS = "0.41.0"
HF_REVISION = "2202fcb69960d94b437e06c19c556a12ceeb57c0"
RES, SEED, ALIGN = 768, 0, 8


def pipeline(tree: Path, snapshot: str, dtype: torch.dtype = torch.float32) -> Any:  # noqa: ANN401 - diffusers pipeline
    """The reference pipeline from upstream's tree and the weights snapshot directory."""
    import diffusers  # noqa: PLC0415 - reference environment only

    assert diffusers.__version__ == DIFFUSERS, diffusers.__version__  # noqa: S101 - tool
    from diffusers.models.controlnets import controlnet  # noqa: PLC0415

    sys.modules.setdefault("diffusers.models.controlnet", controlnet)  # the diffusers 0.28 path of controlnetvae.py
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline  # noqa: PLC0415  # reference tree on sys.path

    return YOSONormalsPipeline.from_pretrained(
        snapshot, trust_remote_code=True, variant="fp16", torch_dtype=dtype, t_start=0, safety_checker=None
    )


def latent_shape(h: int, w: int) -> tuple[int, int, int, int]:
    """The latent shape of an `h x w` input: longest side `RES` (truncated), padded to `ALIGN`, divided by 8."""
    ph, pw = h * RES // max(h, w), w * RES // max(h, w)
    return (1, 4, -(-ph // ALIGN), -(-pw // ALIGN))


def predict(pipe: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - diffusers pipeline
    """`[1, 3, H, W]` float in [0, 1] -> unit normals `[H, W, 3]` in the camera frame (float32)."""
    noise = torch.randn(latent_shape(*x.shape[-2:]), generator=torch.Generator().manual_seed(SEED))
    out = pipe(
        x,
        processing_resolution=RES,
        match_input_resolution=True,
        latents=noise.to(x.device, pipe.unet.dtype),
        output_type="pt",
    )
    return out.prediction[0].permute(1, 2, 0).float() * torch.tensor([-1.0, -1.0, -1.0], device=x.device)


def main() -> None:
    tree = reference_tree(Path(sys.argv[1]), REF_COMMIT)
    pipe = pipeline(tree, str(Path(sys.argv[2])))
    pipe.set_progress_bar_config(disable=True)
    x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
    with torch.inference_mode():
        normals = predict(pipe, x).numpy().astype(np.float32)
    write(
        "stablenormal-turbo@1",
        {"normals": normals},
        family="stablenormal",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=(
            f"CPU float32, weights Stable-X/yoso-normal-v0-3@{HF_REVISION} (fp16 files), t_start 0, processing "
            f"resolution {RES}, CPU generator seed {SEED} as `latents`, camera frame, diffusers {DIFFUSERS}, "
            f"torch {torch.__version__}"
        ),
        task="depth",
    )
    print(f"  normals {normals.shape}, mean z {normals[..., 2].mean():.4f}")


if __name__ == "__main__":
    main()
