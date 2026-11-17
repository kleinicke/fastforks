"""Lotus normals goldens from the reference implementation.

Reference: upstream EnVision-Research/Lotus (`REF_COMMIT`), `pipeline.py` `LotusGPipeline` / `LotusDPipeline`, CPU
float32, `from_pretrained` of the variant's weights snapshot (`HF_REVISIONS`), called as `infer.py` calls it: the image
mapped to [-1, 1], prompt '', one step at timestep 999, the normals task switcher, `processing_res=768`, bilinear
resizing, `match_input_res` (nearest-exact back to the input size), with `infer.py --seed 42`'s seeding (`seed_all`: the
global RNG; the latent generator a CPU `torch.Generator` seeded alike). The output in [0, 1] is mapped to `2 x - 1`,
normalised and stored in the camera frame (x right, y down, z forward): Lotus' (x left, y up, z out) -> (-x, -y, -z).
Run this with a Python that has diffusers (`DIFFUSERS`), transformers, accelerate, torchvision and matplotlib;
upstream's unused `tensorboard` import is stubbed.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/lotus.py LOTUS_CHECKOUT G_SNAPSHOT/ D_SNAPSHOT/
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/EnVision-Research/Lotus"
REF_COMMIT = "b737e12238d2a41fc582ee8c6c6b1104d318d799"
DIFFUSERS = "0.41.0"
HF_REVISIONS = {
    "lotus-normal-g-v1-1": "92b42fd7f5c3f97f09a2af8e96263d87ea899bf1",
    "lotus-normal-d-v1-1": "1d4d52149432060efabb0827e0a9fe4a0b63e817",
}
TIMESTEP, RES, SEED = 999, 768, 42


def pipeline(tree: Path, snapshot: str, ref: str, dtype: torch.dtype = torch.float32) -> Any:  # noqa: ANN401 - diffusers pipeline
    """The reference pipeline of `ref` from upstream's tree and the weights snapshot directory."""
    import diffusers  # noqa: PLC0415 - reference environment only

    assert diffusers.__version__ == DIFFUSERS, diffusers.__version__  # noqa: S101 - tool
    import accelerate  # noqa: F401, PLC0415 - probes for tensorboard on import: before the stub below

    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    sys.modules["tensorboard"] = types.ModuleType("tensorboard")  # imported by upstream, never used
    try:
        import pipeline as lotus  # noqa: PLC0415  # reference tree on sys.path
    finally:
        del sys.modules["tensorboard"]

    cls = lotus.LotusGPipeline if ref.startswith("lotus-normal-g") else lotus.LotusDPipeline
    return cls.from_pretrained(snapshot, torch_dtype=dtype)


def predict(pipe: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - diffusers pipeline
    """`[1, 3, H, W]` float in [0, 1] -> unit normals `[H, W, 3]` in the camera frame (float32)."""
    task = torch.tensor([1, 0]).float().unsqueeze(0).to(x.device)
    task = torch.cat([torch.sin(task), torch.cos(task)], dim=-1)
    torch.manual_seed(SEED)  # infer.py seed_all: the posterior sample draws from the global RNG
    out = pipe(
        rgb_in=x * 2 - 1,
        prompt="",
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(SEED),
        output_type="pt",
        timesteps=[TIMESTEP],
        task_emb=task.to(pipe.unet.dtype),
        processing_res=RES,
        match_input_res=True,
        resample_method="bilinear",
    ).images[0]
    n = out.float() * 2 - 1
    n = n / torch.linalg.vector_norm(n, dim=0, keepdim=True).clamp(min=1e-6)
    return n.permute(1, 2, 0) * torch.tensor([-1.0, -1.0, -1.0], device=n.device)  # x left, y up, z to the viewer


def main() -> None:
    tree = reference_tree(Path(sys.argv[1]), REF_COMMIT)
    x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
    for ref, snapshot in zip(HF_REVISIONS, sys.argv[2:4], strict=True):
        pipe = pipeline(tree, snapshot, ref)
        pipe.set_progress_bar_config(disable=True)
        normals = predict(pipe, x).numpy().astype(np.float32)
        write(
            f"{ref}@1",
            {"normals": normals},
            family="lotus",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=(
                f"CPU float32, weights jingheya/{ref}@{HF_REVISIONS[ref]}, diffusers {DIFFUSERS}, timestep {TIMESTEP}, "
                f"processing resolution {RES}, seed {SEED} (global RNG and CPU generator), camera frame, "
                f"torch {torch.__version__}"
            ),
            task="depth",
        )
        print(f"  normals {normals.shape}, mean z {normals[..., 2].mean():.4f}")


if __name__ == "__main__":
    main()
