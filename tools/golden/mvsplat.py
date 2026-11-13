"""MVSplat golden from the reference implementation.

Reference: upstream donydchen/mvsplat (`REF_COMMIT`), CPU float32, `EncoderCostVolume.forward` (eval, the released
`re10k.ckpt` config: `config/model/encoder/costvolume.yaml` with `config/experiment/re10k.yaml`'s overrides,
`num_context_views` = the number of views) on the 4 views of the `tests/synth.py` scene, with the runner's pre- and
post-processing: images resized to long side `LONG_SIDE` (multiples of 16, bicubic antialiased, clamped), intrinsics
normalised by the input size (pixel centres at `(i + 0.5) / W`), near / far from the scene's depth range, the scene
scaled to near = 1 (translations, near, far; depth, centres and scales scaled back), depth and confidence (the largest
softmax weight over the depth candidates, as upstream's refinement input) resized back (bilinear).

Upstream returns world covariances and the camera-frame rotations (`visualization_dump`); the golden rotations are
`q(R_c2w) q_cam` (`w x y z`, `w >= 0`) from Shepperd's method in float64, checked here against upstream's covariance.

    FASTFORKS_GOLDEN_PATH=... python -I tools/golden/mvsplat.py MVSPLAT_CHECKOUT CKPT EXTRA_SITE

`EXTRA_SITE`: a directory with `einops`, `jaxtyping`, `e3nn` and `omegaconf` (upstream imports them), e.g. a
`uv pip install --target`.
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
from depthsplat import hamilton, shepperd

from synth import scene

UPSTREAM = "https://github.com/donydchen/mvsplat"
REF_COMMIT = "01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01"
LONG_SIDE = 256
PACKAGES = ("src", "src.model", "src.model.encoder", "src.dataset")  # their __init__ pull in training code


def network(repo: Path, ckpt: Path, views: int) -> torch.nn.Module:
    """`EncoderCostVolume` with the released config from the checkout at `REF_COMMIT`, CPU float32, eval."""
    tree = reference_tree(repo, REF_COMMIT)
    for name in PACKAGES:
        m = types.ModuleType(name)
        m.__path__ = [str(tree / name.replace(".", "/"))]
        sys.modules[name] = m
    # ruff: noqa: PLC0415 - reference checkout
    from omegaconf import OmegaConf
    from src.global_cfg import set_cfg
    from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
    from src.model.encoder.encoder_costvolume import EncoderCostVolume, EncoderCostVolumeCfg, OpacityMappingCfg
    from src.model.encoder.visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

    set_cfg(OmegaConf.create({"mode": "test", "dataset": {"view_sampler": {"num_context_views": views}}}))
    cfg = EncoderCostVolumeCfg(
        name="costvolume", d_feature=128, num_depth_candidates=128, num_surfaces=1,
        visualizer=EncoderVisualizerCostVolumeCfg(num_samples=8, min_resolution=256, export_ply=False),
        gaussian_adapter=GaussianAdapterCfg(gaussian_scale_min=0.5, gaussian_scale_max=15.0, sh_degree=4),
        opacity_mapping=OpacityMappingCfg(initial=0.0, final=0.0, warm_up=1), gaussians_per_pixel=1,
        unimatch_weights_path=None, downscale_factor=4, shim_patch_size=4, multiview_trans_attn_split=2,
        costvolume_unet_feat_dim=128, costvolume_unet_channel_mult=[1, 1, 1], costvolume_unet_attn_res=[4],
        depth_unet_feat_dim=32, depth_unet_attn_res=[16], depth_unet_channel_mult=[1, 1, 1, 1, 1],
        wo_depth_refine=False, wo_cost_volume=False, wo_backbone_cross_attn=False, wo_cost_volume_refine=False,
        use_epipolar_trans=False,
    )  # fmt: skip
    enc = EncoderCostVolume(cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    enc.load_state_dict({k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}, strict=True)
    return enc.eval()


def infer(
    enc: torch.nn.Module, x: torch.Tensor, cameras: tuple[torch.Tensor, torch.Tensor, float, float], long_side: int
) -> tuple[object, dict[str, torch.Tensor], torch.Tensor, float]:
    """Upstream forward (b = 1) on `x` `[V, 3, H, W]` (0-1) with the runner's pre-processing.

    `cameras`: pixel intrinsics `[V, 3, 3]`, cam_to_world `[V, 4, 4]`, near, far. Returns the Gaussians, the
    visualization dump (network-scale depth, scales, camera-frame rotations), the candidate confidence at the network
    resolution `[V, 1, h, w]` and the scene scale.
    """
    k_px, c2w, near, far = cameras
    v, _, h0, w0 = x.shape
    sc = 1.0 / near  # the runner's scene scale (near = 1)
    net_c2w = c2w.float().clone()
    net_c2w[:, :3, 3] *= sc
    f = long_side / max(h0, w0)
    h, w = (max(16, round(d * f / 16) * 16) for d in (h0, w0))
    x = F.interpolate(x, (h, w), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    k = k_px.double().clone()
    k[:, :2, 2] += 0.5
    k[:, 0] /= w0
    k[:, 1] /= h0
    near_t, far_t = (torch.full((1, v), d * sc, device=x.device) for d in (near, far))
    for m in enc.modules():  # the UNets' cross-view attention, built for `num_context_views` (the golden's V)
        if hasattr(m, "n_frames"):
            m.n_frames = v
    logits = []
    hook = enc.depth_predictor.depth_head_lowres.register_forward_hook(lambda _m, _i, o: logits.append(o))
    context = {"image": x[None], "intrinsics": k.float()[None], "extrinsics": net_c2w[None], "near": near_t,
               "far": far_t}  # fmt: skip
    dump: dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            g = enc(context, global_step=0, deterministic=True, visualization_dump=dump)
    finally:
        hook.remove()
    prob = F.interpolate(F.softmax(logits[0].float(), dim=1).max(dim=1, keepdim=True)[0], scale_factor=4)
    return g, dump, prob, sc


def predict(enc: torch.nn.Module, s: dict[str, np.ndarray], long_side: int) -> dict[str, np.ndarray]:
    """Upstream forward on the scene `s` with the runner's pre/post-processing (float64 world rotations)."""
    img = torch.from_numpy(s["images"]).permute(0, 3, 1, 2).float() / 255
    h0, w0 = img.shape[2:]
    c2w = torch.from_numpy(s["cam_to_world"]).float()
    near, far = (float(s["depth_range"][i]) for i in (0, 1))
    k = torch.from_numpy(s["intrinsics"]).double()
    g, dump, prob, sc = infer(enc, img, (k, c2w, near, far), long_side)
    h, w = prob.shape[2:]
    depth = dump["depth"][0, ..., 0, 0] / sc  # [V, h, w]

    def up(t: torch.Tensor) -> np.ndarray:
        return F.interpolate(t.float(), (h0, w0), mode="bilinear", align_corners=False)[:, 0].numpy()

    scales = (dump["scales"][0] / sc).double().numpy()
    q_cam = dump["rotations"][0].double().numpy()[:, [3, 0, 1, 2]]
    q_cam /= np.linalg.norm(q_cam, axis=1, keepdims=True)
    q_c2w = np.repeat(shepperd(c2w[:, :3, :3].double().numpy()), h * w, 0)
    q = hamilton(q_c2w, q_cam)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    from src.model.encoder.common.gaussians import build_covariance

    cov = build_covariance(torch.from_numpy(scales), torch.from_numpy(q[:, [1, 2, 3, 0]])).numpy()
    ref = (g.covariances[0] / sc**2).double().numpy()
    err = np.abs(cov - ref).max()
    print(f"  world rotations vs upstream covariance: max |diff| {err:.3g} (|cov| max {np.abs(cov).max():.3g})")
    if err >= 1e-5 * max(1.0, np.abs(cov).max()):
        msg = f"world rotations disagree with upstream's covariance: {err:.3g}"
        raise RuntimeError(msg)
    return {
        "depth": up(depth[:, None]),
        "confidence": up(prob),
        "gaussians.means": (g.means[0] / sc).numpy(),
        "gaussians.scales": scales,
        "gaussians.rotations": q,
        "gaussians.harmonics": g.harmonics[0].numpy(),
        "gaussians.opacities": g.opacities[0].numpy(),
    }


def main() -> None:
    sys.path.append(sys.argv[3])
    s = scene()
    enc = network(Path(sys.argv[1]), Path(sys.argv[2]), len(s["images"]))
    maps = predict(enc, s, LONG_SIDE)
    h0, w0 = s["images"].shape[1:3]
    h, w = (max(16, round(x * LONG_SIDE / max(h0, w0) / 16) * 16) for x in (h0, w0))
    write(
        "mvsplat-re10k@1",
        {k: np.asarray(v, np.float32) for k, v in maps.items()},
        family="mvsplat",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, synth scene (4 views) at {h0}x{w0} -> network {h}x{w}, torch {torch.__version__}",
        task="multiview",
    )
    gt, d = s["depth"], maps["depth"]
    print(f"  depth AbsRel {np.mean(np.abs(d - gt) / gt):.4f}, {len(maps['gaussians.means'])} gaussians")


if __name__ == "__main__":
    main()
