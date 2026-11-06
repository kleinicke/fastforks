"""C-FoundationStereo goldens from the reference implementation.

Reference: NVIDIA TAO PyTorch (`tao_pytorch_backend`, Apache-2.0) at `REF_COMMIT`, its own FoundationStereo /
FastFoundationStereo modules with TAO's model config (`DepthNetModelConfig` defaults for C-FS-S, the model section of
`experiment_fast_foundation_stereo.yaml` for C-FFS), CPU float32, `valid_iters` iterations, inputs ImageNet-normalised
and padded as TAO's `pl_stereo_model` does (`InputPadder(divis_by=32)`, sintel mode). TAO imports timm, scipy,
torchvision, PIL and omegaconf, and the checkpoints are read with fastforks: run this with the fastforks environment
plus a site-packages that has them appended to `sys.path` (same torch build; we used an environment that has them).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/c_foundationstereo.py TAO_CHECKOUT WEIGHTS_DIR [c-fs-s c-ffs ffs]

`WEIGHTS_DIR` holds the two Hugging Face files (`trainable_foundation_stereo_small.pth`,
`model_best_bp2_serialize.pth`; `ffs` also needs the research checkpoint as
`23-36-37/model_best_bp2_serialize.pth`).
Only the model modules of TAO run: its package `__init__`s (ONNX / SDPA patches,
Lightning, the backbone registry with fvcore) are replaced by empty package shells and five stubs (`_setup`). The
checkpoints are pickles; they are read with fastforks' restricted loader (the same globals table as the adapter),
then C-FFS gets TAO's own key remap (`ckpt_utils`), as TAO's `load_ffs_pretrained` does after its plain `torch.load`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import ROOT, pair, reference_tree, write

sys.path.insert(0, str(ROOT / "src"))

from fastforks._core.checkpoint import load_state_dict
from fastforks.stereo.models.c_foundationstereo.model import _ffs_extra

UPSTREAM = "https://github.com/NVIDIA/tao_pytorch_backend"
REF_COMMIT = "77fb1a4bca02de3ea43de333a5822d57694b6b22"
VARIANTS = {"c-fs-s": "c-foundationstereo-s", "c-ffs": "c-fast-foundationstereo", "ffs": "fast-foundationstereo"}
FILES = {
    "c-fs-s": "trainable_foundation_stereo_small.pth",
    "c-ffs": "model_best_bp2_serialize.pth",
    "ffs": "23-36-37/model_best_bp2_serialize.pth",  # research checkpoint, same network
}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # TAO DepthNetAugmentationConfig input_mean / input_std
PKG = "nvidia_tao_pytorch"


def _setup(root: Path) -> None:
    """TAO's model modules without its package `__init__`s; stubs for the training-only imports."""
    sys.path.insert(0, str(root))

    def pkg(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        m.__path__ = [str(root.joinpath(*name.split(".")))]
        sys.modules[name] = m
        if "." in name:
            parent, _, child = name.rpartition(".")
            setattr(pkg(parent), child, m)
        return m

    def stub(name: str, **attrs: object) -> None:
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        parent, _, child = name.rpartition(".")
        setattr(pkg(parent), child, m)

    class Registry(dict):
        """The fvcore backbone registry: `register` only."""

        def register(self, obj: object = None) -> object:
            def deco(f: object) -> object:
                self[getattr(f, "__name__", str(f))] = f
                return f

            return deco(obj) if obj is not None else deco

    for p in ("config", "core.utils", "cv.depth_net.model.mono_depth", "cv.depth_net.model.stereo_depth"):
        pkg(f"{PKG}.{p}")
    pkg(f"{PKG}.cv.backbone_v2").BACKBONE_REGISTRY = Registry()  # module shell
    stub(f"{PKG}.cv.depth_net.model.mono_depth.pl_mono_model", MonoDepthNetPlModel=object)  # Lightning wrapper
    stub(f"{PKG}.core.distributed.comm", get_global_rank=lambda: 0)
    stub(f"{PKG}.core.loggers.api_logging", status_logging=None)  # logging only (needs Lightning)
    stub(f"{PKG}.core.utils.ptm_utils", load_pretrained_weights=None)  # training init only


def _config(root: Path, name: str) -> Any:  # noqa: ANN401 - an OmegaConf DictConfig
    from nvidia_tao_pytorch.config.depth_net.model import DepthNetModelConfig  # noqa: PLC0415  # reference tree on sys.path
    from omegaconf import OmegaConf  # noqa: PLC0415  # reference-only dependency

    cfg = OmegaConf.structured(DepthNetModelConfig)
    if name == "c-fs-s":
        cfg.encoder = "vits"  # C-FS-S: DINOv2 ViT-S, the rest TAO's defaults
    else:
        spec = OmegaConf.load(root / PKG / "cv/depth_net/experiment_specs/experiment_fast_foundation_stereo.yaml")
        cfg = OmegaConf.merge(cfg, spec.model)
    cfg.mixed_precision = False
    return cfg


def _net(root: Path, name: str, weights: Path) -> tuple[torch.nn.Module, int]:
    cfg = _config(root, name)
    path = weights / FILES[name]
    if name == "c-fs-s":
        from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.foundation_stereo import (  # noqa: PLC0415  # reference tree
            FoundationStereo,
        )

        net = FoundationStereo(cfg)
        net.load_state_dict(load_state_dict(path, key="state_dict", strip="model."))  # Lightning checkpoint
    else:
        from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.fast_foundation_stereo import ckpt_utils as cu  # noqa: PLC0415  # reference tree
        from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.fast_foundation_stereo.fast_foundation_stereo import (  # noqa: PLC0415  # reference tree
            FastFoundationStereo,
        )

        net = FastFoundationStereo(cfg)
        sd = cu._strip_prefix(load_state_dict(path, strip="", extra=_ffs_extra()), prefix="module.")  # noqa: SLF001
        sd = cu._apply_substring_remap(cu._apply_prefix_remap(sd, cu._PREFIX_REMAP_RULES), cu._SUBSTRING_REMAP_RULES)  # noqa: SLF001
        res = net.load_state_dict(sd, strict=False)  # as load_ffs_pretrained, which asserts no missing / unexpected
        missing = [k for k in res.missing_keys if k not in cu._OPTIONAL_MISSING_KEYS]  # noqa: SLF001
        if missing or res.unexpected_keys:
            raise RuntimeError((missing, res.unexpected_keys))
    return net.eval(), int(cfg.valid_iters)


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    root = reference_tree(repo, REF_COMMIT)
    _setup(root)
    from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.utils import InputPadder  # noqa: PLC0415  # reference tree

    mean, std = torch.tensor(MEAN)[:, None, None], torch.tensor(STD)[:, None, None]
    left, right = ((torch.from_numpy(x).permute(2, 0, 1).float() / 255 - mean) / std for x in pair())
    padder = InputPadder(left[None].shape, divis_by=32)
    lp, rp = padder.pad(left[None], right[None])
    for name in sys.argv[3:] or VARIANTS:
        net, iters = _net(root, name, weights)
        with torch.inference_mode():
            disp = net(lp, rp, iters=iters, test_mode=True)
        disp = padder.unpad(disp.float())[0, 0].numpy()
        write(
            f"{VARIANTS[name]}@1",
            {"disparity": disp},
            family="c_foundationstereo",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, {iters} iters, TAO package shells + 5 import stubs, torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
