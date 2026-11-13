"""AnySplat golden from the reference implementation.

Reference: upstream InternRobotics/AnySplat (`REF_COMMIT`), CPU float32, the inference part of
`EncoderAnySplat.forward` (released config: depth head, no pruning, opacity mapping the identity; `voxelize` off, the
golden's `voxel_size = 0`) on the 4 views of the `tests/synth.py` scene, built from upstream's modules (VGGT copy,
`VGGT_DPT_GS_Head`, `UnifiedGaussianAdapter`) without the training-side imports of `anysplat.py`, with the runner's
pre- and post-processing: images resized to long side `LONG_SIDE` (multiples of 14, bicubic antialiased, clamped),
maps resized back (bilinear), intrinsics rescaled to the input pixels (principal point -0.5), `cam_to_world = inv(w2c)`,
quaternions x y z w -> w x y z. Upstream runs the aggregator under bf16 autocast (CUDA); here all in float32.

    FASTFORKS_GOLDEN_PATH=... python -I tools/golden/anysplat.py ANYSPLAT_CHECKOUT model.safetensors [EXTRA_SITE]

`EXTRA_SITE`: a directory with `einops` and `jaxtyping` (upstream imports them), e.g. a `uv pip install --target`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "src")]  # python -I
# ruff: noqa: E402 - imports after the path setup

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write
from safetensors.torch import load_file
from torch import nn

from synth import scene

UPSTREAM = "https://github.com/InternRobotics/AnySplat"
REF_COMMIT = "5f5e208a7dd57d52e43ea0d553a95eab526e8775"
LONG_SIDE = 224
LAYERS = [4, 11, 17, 23]


def _packages(tree: Path) -> None:
    """Empty parent packages (their `__init__` pulls in training code) and an empty `xformers.ops` (unused here)."""
    for name in ("src", "src.model", "src.model.encoder", "src.model.encoder.heads", "src.model.encoder.common"):
        m = types.ModuleType(name)
        m.__path__ = [str(tree / name.replace(".", "/"))]
        sys.modules[name] = m
    for name in ("xformers", "xformers.ops", "src.geometry", "src.geometry.projection", "src.misc"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["src.geometry.projection"].get_world_rays = None
    sh = sys.modules["src.misc.sh_rotation"] = types.ModuleType("src.misc.sh_rotation")
    sh.rotate_sh = None


def network(repo: Path, ckpt: Path) -> types.SimpleNamespace:
    """`EncoderAnySplat`'s inference modules from the checkout at `REF_COMMIT`, on the CPU in float32.

    Returns:
        `enc` (aggregator, camera_head, depth_head, gaussian_param_head), `adapter` (`UnifiedGaussianAdapter`) and the
        upstream helpers `unproject` and `pose` that `predict` uses.
    """
    tree = reference_tree(repo, REF_COMMIT)
    _packages(tree)  # the checkout is reached only through these packages: it is not on sys.path
    # ruff: noqa: PLC0415 - reference checkout
    from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg, UnifiedGaussianAdapter
    from src.model.encoder.heads.vggt_dpt_gs_head import VGGT_DPT_GS_Head
    from src.model.encoder.vggt.heads.camera_head import CameraHead
    from src.model.encoder.vggt.heads.dpt_head import DPTHead
    from src.model.encoder.vggt.models.aggregator import Aggregator
    from src.model.encoder.vggt.utils.geometry import batchify_unproject_depth_map_to_point_map
    from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri

    enc = nn.Module()  # EncoderAnySplat's inference modules (VGGT(): img_size 518, patch 14, embed 1024)
    enc.aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
    enc.camera_head = CameraHead(dim_in=2048)
    enc.depth_head = DPTHead(dim_in=2048, output_dim=2, activation="exp", conf_activation="expp1")
    adapter = UnifiedGaussianAdapter(GaussianAdapterCfg(gaussian_scale_min=0.5, gaussian_scale_max=15.0, sh_degree=4))
    enc.gaussian_param_head = VGGT_DPT_GS_Head(
        dim_in=2048, patch_size=(14, 14), output_dim=1 + adapter.d_in + 1, activation="norm_exp",
        conf_activation="expp1", features=256,
    )  # fmt: skip
    sd = {k.removeprefix("encoder."): v for k, v in load_file(ckpt).items()}  # bf16 + fp32: cast on copy
    enc.load_state_dict(sd, strict=True)
    return types.SimpleNamespace(
        enc=enc.eval(), adapter=adapter, unproject=batchify_unproject_depth_map_to_point_map,
        pose=pose_encoding_to_extri_intri,
    )  # fmt: skip


def voxelize(feats: torch.Tensor, pts: torch.Tensor, conf: torch.Tensor, voxel: float) -> tuple[torch.Tensor, ...]:
    """Upstream `voxelizaton_with_fusion` on flat `[N, C]` / `[N, 3]` / `[N]`.

    `torch_scatter`'s `scatter_max` / `scatter_add` are written with `scatter_reduce` / `index_add_` (the same sums).
    """
    inv = torch.unique((pts / voxel).round().int(), dim=0, return_inverse=True)[1]
    n = int(inv.max()) + 1

    def add(x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros((n, *x.shape[1:])).index_add_(0, inv, x)

    cmax = conf.new_full((n,), -torch.inf).scatter_reduce(0, inv, conf, "amax")
    e = torch.exp(conf - cmax[inv])
    w = (e / (add(e)[inv] + 1e-6))[:, None]
    return add(pts * w), add(feats * w)


def predict(
    net: types.SimpleNamespace, img: torch.Tensor, long_side: int, voxel: float, *, bf16: bool = False
) -> dict[str, torch.Tensor]:
    """`EncoderAnySplat.forward` (b = 1, eval) on `img` `[V, 3, H, W]` in 0-1 with the runner's pre/post-processing.

    `bf16`: upstream's own precision (the aggregator under `torch.autocast(bfloat16)`, the heads in float32); `voxel`
    0 keeps one Gaussian per pixel.
    """
    h0, w0 = img.shape[-2:]
    f = long_side / max(h0, w0)
    h, w = (max(14, round(v * f / 14) * 14) for v in (h0, w0))
    image = F.interpolate(img, (h, w), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)[None]
    enc, adapter = net.enc, net.adapter
    with torch.no_grad():
        with torch.autocast(img.device.type, torch.bfloat16, enabled=bf16):
            tokens, start = enc.aggregator(image.to(torch.bfloat16) if bf16 else image, intermediate_layer_idx=LAYERS)
        with torch.autocast(img.device.type, enabled=False):
            extrinsic, intrinsic = net.pose(enc.camera_head(tokens)[-1], image.shape[-2:])
            depth_map, depth_conf = enc.depth_head(tokens, images=image, patch_start_idx=start)
            pts_all = net.unproject(depth_map, extrinsic, intrinsic)
        out = enc.gaussian_param_head(
            tokens, pts_all.flatten(0, 1).permute(0, 3, 1, 2), image, patch_start_idx=start, image_size=(h, w)
        )
        d = 1 + adapter.d_in
        feats, pts = out[0, :, :d].permute(0, 2, 3, 1).reshape(-1, d), pts_all[0].reshape(-1, 3)
        if voxel > 0:
            pts, feats = voxelize(feats, pts, out[0, :, d].flatten(), voxel)
        opacity = feats[..., 0].sigmoid()  # map_pdf_to_opacity with initial = final = 0: the identity
        g = adapter.forward(pts, pts[..., -1:], opacity, feats[..., 1:])

    def up(t: torch.Tensor) -> torch.Tensor:
        return F.interpolate(t.float(), (h0, w0), mode="bilinear", align_corners=False)

    ext, k = extrinsic[0], intrinsic[0].clone()
    k[:, 0] *= w0 / w
    k[:, 1] *= h0 / h
    k[:, :2, 2] -= 0.5
    w2c = torch.eye(4, device=ext.device).repeat(ext.shape[0], 1, 1)
    w2c[:, :3] = ext
    return {
        "depth": up(depth_map[0].permute(0, 3, 1, 2))[:, 0],
        "confidence": up(depth_conf[0][:, None])[:, 0],
        "world_points": up(pts_all[0].permute(0, 3, 1, 2)).permute(0, 2, 3, 1),
        "intrinsics": k,
        "cam_to_world": torch.linalg.inv(w2c),
        "gaussians.means": g.means,
        "gaussians.scales": g.scales,
        "gaussians.rotations": g.rotations[:, [3, 0, 1, 2]],
        "gaussians.harmonics": g.harmonics,
        "gaussians.opacities": g.opacities,
    }


def main() -> None:
    if len(sys.argv) > 3:  # noqa: PLR2004
        sys.path.append(sys.argv[3])
    net = network(Path(sys.argv[1]), Path(sys.argv[2]))
    s = scene()
    img = torch.from_numpy(s["images"]).permute(0, 3, 1, 2).float() / 255
    h0, w0 = img.shape[-2:]
    out = predict(net, img, LONG_SIDE, 0.0)
    maps = {n: v.numpy() for n, v in out.items()}
    h, w = (max(14, round(v * LONG_SIDE / max(h0, w0) / 14) * 14) for v in (h0, w0))
    write(
        "anysplat@1",
        maps,
        family="anysplat",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, synth scene (4 views) at {h0}x{w0} -> network {h}x{w}, voxel_size 0, torch "
        f"{torch.__version__}",
        task="multiview",
    )
    gt, d = s["depth"], maps["depth"]
    sc = np.median(gt) / np.median(d)
    n = len(maps["gaussians.means"])
    print(f"  depth AbsRel after median scaling {np.mean(np.abs(d * sc - gt) / gt):.4f}, {n} gaussians")


if __name__ == "__main__":
    main()
