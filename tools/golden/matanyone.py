"""MatAnyone video matting goldens from the reference implementation.

Reference: upstream pq-yang/MatAnyone (`REF_COMMIT`), unmodified: `MatAnyone(cfg, single_object=True)` with the
config `eval_matanyone_config.yaml` + `model/base.yaml` (read here with PyYAML; hydra composes the same two files),
`load_weights`, and `InferenceCore.process_video`'s loop on frames in memory instead of a video file: the mask closed
with `gen_dilate` / `gen_erosion` (cv2, radius 10), 10 warm-up copies of frame 0, `step` per frame, the alpha
`output_prob_to_mask` of each real frame (float, without the uint8 rounding of the written video). CPU float32
(process_video's autocast is CUDA fp16 on a GPU and CPU bfloat16 without one; the reference runs without it).
Inputs: `tests/synth.py clip()` (6 frames) and the disk mask of `clip_prompts()["video_mask_"]` (0 / 255) stored as
`input` (frame 0) and `mask`; outputs `alpha` (frame 0 alone as a video) and `video_alpha` (the clip).

    FASTFORKS_GOLDEN_PATH=... CUDA_VISIBLE_DEVICES= python tools/golden/matanyone.py MATANYONE_CHECKOUT WEIGHTS

`omegaconf` (annotations; `OmegaConf.to_container` for the Hub mixin's config record) and `imageio` (video
writing) are stub modules.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml
from _common import reference_tree, write

from synth import clip, clip_prompts

FAMILY = "matanyone"
UPSTREAM = "https://github.com/pq-yang/MatAnyone"
REF_COMMIT = "e5ddc534c1fff9bb9e54cf476095d29071b7cb4f"
WARMUP, R_DILATE, R_ERODE = 10, 10, 10  # process_video defaults


class Config(dict):
    """OmegaConf-like view of the YAML: attribute access (missing: AttributeError) and items."""

    def __getattr__(self, key: str) -> object:
        """`cfg.key` as `cfg["key"]`."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


def _config(d: object) -> object:
    return Config({k: _config(v) for k, v in d.items()}) if isinstance(d, dict) else d


def load_upstream(checkout: Path) -> Path:
    """Put the pinned tree on `sys.path` with the stub modules; returns its root."""
    root = reference_tree(checkout, REF_COMMIT)
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.__dict__.update({"DictConfig": dict, "OmegaConf": types.SimpleNamespace(to_container=dict)})
    sys.modules["omegaconf"] = omegaconf
    sys.modules["imageio"] = types.ModuleType("imageio")
    sys.path.insert(0, str(root))
    return root


def config(root: Path) -> Config:
    """`eval_matanyone_config.yaml` with `model/base.yaml` as `model` (hydra defaults), no ImageNet download."""
    cfg = yaml.safe_load((root / "matanyone/config/eval_matanyone_config.yaml").read_text())
    cfg["model"] = yaml.safe_load((root / "matanyone/config/model/base.yaml").read_text())
    cfg["model"]["pretrained_resnet"] = False  # the checkpoint replaces the ImageNet weights

    def resolve(v: object) -> object:  # OmegaConf interpolation `${model.embed_dim}` (whole values only)
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            node: object = cfg
            for key in v[2:-1].split("."):
                node = node[key]
            return resolve(node)
        return {k: resolve(x) for k, x in v.items()} if isinstance(v, dict) else v

    return Config({k: _config(resolve(v)) for k, v in cfg.items()})


def build(root: Path, weights: Path) -> torch.nn.Module:
    """Upstream network from the checkpoint (`load_weights`; every key checked: upstream loads non-strict)."""
    from matanyone.model.matanyone import MatAnyone  # noqa: PLC0415 - reference tree on sys.path

    net = MatAnyone(config(root), single_object=True).eval()
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    if set(sd) != set(net.state_dict()):
        msg = f"checkpoint keys differ from the network: {sorted(set(sd) ^ set(net.state_dict()))[:5]}"
        raise SystemExit(msg)
    net.load_weights(sd)
    return net


def close(mask: np.ndarray, dilate: int = R_DILATE, erode: int = R_ERODE) -> torch.Tensor:
    """`process_video`'s mask closing (`gen_dilate` then `gen_erosion`, cv2) as a float 0 / 255 tensor."""
    from matanyone.utils.inference_utils import gen_dilate, gen_erosion  # noqa: PLC0415 - reference tree on sys.path

    if dilate > 0:
        mask = gen_dilate(mask, dilate, dilate)
    if erode > 0:
        mask = gen_erosion(mask, erode, erode)
    return torch.from_numpy(mask).float()


def run(net: torch.nn.Module, vframes: torch.Tensor, m: torch.Tensor, warmup: int = WARMUP) -> list[torch.Tensor]:
    """`process_video`'s loop on `[T, 3, H, W]` 0..255 frames (each moved to `m`'s device) and the closed mask."""
    from matanyone.inference.inference_core import InferenceCore  # noqa: PLC0415 - reference tree on sys.path

    core = InferenceCore(net, cfg=net.cfg, device=m.device)
    vframes = torch.cat([vframes[:1].repeat(warmup, 1, 1, 1), vframes]).float()
    out = []
    with torch.inference_mode():
        for ti in range(len(vframes)):
            image = vframes[ti].to(m.device) / 255.0
            if ti == 0:
                core.step(image, m, objects=[1])
                prob = core.step(image, first_frame_pred=True)
            else:
                prob = core.step(image, first_frame_pred=ti <= warmup)
            if ti >= warmup:
                out.append(core.output_prob_to_mask(prob).float())
    return out


def matte(net: torch.nn.Module, frames: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """`process_video` on `[H, W, 3]` uint8 frames and a `[H, W]` uint8 mask -> alpha `[T, H, W]` float32 (CPU)."""
    vframes = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    return torch.stack(run(net, vframes, close(mask))).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of pq-yang/MatAnyone")
    ap.add_argument("weights", type=Path, help="matanyone.pth (v1.0.0 release)")
    a = ap.parse_args()
    if torch.cuda.is_available():
        msg = "run with CUDA_VISIBLE_DEVICES= (the reference runs on the CPU)"
        raise SystemExit(msg)
    root = load_upstream(a.checkout)
    net = build(root, a.weights)
    frames = clip()
    mask = clip_prompts()["video_mask_"]["input_masks"].astype(np.uint8) * 255
    alpha = matte(net, frames[:1], mask)[0]
    video = matte(net, frames, mask)
    write(
        "matanyone@1",
        {"input": frames[0].astype(np.float32), "mask": mask.astype(np.float32), "alpha": alpha, "video_alpha": video},
        family=FAMILY,
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, process_video loop on frames in memory (mask gen_dilate / gen_erosion 10, 10 warm-up "
        f"frames, float alpha without uint8 rounding); tests/synth.py clip() + disk mask; torch {torch.__version__}",
        task="matting",
    )
    print(f"  matanyone: alpha {alpha.shape} mean {alpha.mean():.4f}, video {video.shape} mean {video.mean():.4f}")


if __name__ == "__main__":
    main()
