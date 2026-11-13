"""DepthSplat golden from the reference implementation.

Reference: upstream cvg/depthsplat (`REF_COMMIT`), CPU float32, `EncoderDepthSplat.forward` (eval, the released
`depthsplat-gs-small-re10k-256x256-view2` config: `config/model/encoder/depthsplat.yaml`) on the 4 views of the
`tests/synth.py` scene, with the runner's pre- and post-processing: images resized to long side `LONG_SIDE` (multiples
of 16, bicubic antialiased, clamped), intrinsics normalised by the input size (pixel centres at `(i + 0.5) / W`), near /
far from the scene's depth range, the scene scaled to near = 1 (translations, near, far; depth, centres and scales
scaled back), depth and confidence (the largest match probability, as upstream's Gaussian head input) resized back
(bilinear). Upstream's ViT-S comes from `torch.hub.load("facebookresearch/dinov2", ...)`; here the
same hub code from a local copy (`DINOV2_HUB`, `pretrained=False`: the checkpoint holds the weights).

Upstream returns world covariances and the camera-frame rotations (`visualization_dump`); the golden rotations are
`q(R_c2w) q_cam` (`w x y z`, `w >= 0`) from Shepperd's method in float64, checked here against upstream's covariance.

    FASTFORKS_GOLDEN_PATH=... python -I tools/golden/depthsplat.py DEPTHSPLAT_CHECKOUT CKPT DINOV2_HUB EXTRA_SITE

`EXTRA_SITE`: a directory with `einops`, `jaxtyping` and `e3nn` (upstream imports them), e.g. a `uv pip install
--target`.
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

from synth import scene

UPSTREAM = "https://github.com/cvg/depthsplat"
REF_COMMIT = "2dad25a7b9ba7c5537aba08463ef88d8511a06be"
LONG_SIDE = 256
PACKAGES = (
    "src", "src.model", "src.model.encoder", "src.model.encoder.common", "src.model.encoder.unimatch",
    "src.model.encoder.visualization", "src.geometry", "src.misc", "src.dataset", "src.dataset.shims",
)  # fmt: skip


def network(repo: Path, ckpt: Path, hub: Path) -> torch.nn.Module:
    """`EncoderDepthSplat` with the released config from the checkout at `REF_COMMIT`, CPU float32, eval."""
    tree = reference_tree(repo, REF_COMMIT)
    for name in PACKAGES:  # empty parents: `src.model.encoder` / `src.dataset` pull in training code and datasets
        m = types.ModuleType(name)
        m.__path__ = [str(tree / name.replace(".", "/"))]
        sys.modules[name] = m
    load = torch.hub.load
    torch.hub.load = lambda _repo, name, **_kw: load(str(hub), name, source="local", pretrained=False)
    # ruff: noqa: PLC0415 - reference checkout
    from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
    from src.model.encoder.encoder_depthsplat import EncoderDepthSplat, EncoderDepthSplatCfg
    from src.model.encoder.visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg

    cfg = EncoderDepthSplatCfg(
        name="depthsplat", d_feature=128, num_depth_candidates=128, num_surfaces=1,
        visualizer=EncoderVisualizerDepthSplatCfg(num_samples=8, min_resolution=256, export_ply=False),
        gaussian_adapter=GaussianAdapterCfg(gaussian_scale_min=1e-10, gaussian_scale_max=3.0, sh_degree=2),
        gaussians_per_pixel=1, unimatch_weights_path=None, downscale_factor=4, shim_patch_size=4,
        multiview_trans_attn_split=2, costvolume_unet_feat_dim=128, costvolume_unet_channel_mult=[1, 1, 1],
        costvolume_unet_attn_res=[], depth_unet_feat_dim=64, depth_unet_attn_res=[], depth_unet_channel_mult=[1, 1, 1],
        num_scales=1, upsample_factor=4, lowest_feature_resolution=4, depth_unet_channels=128,
        grid_sample_disable_cudnn=False, large_gaussian_head=False, color_large_unet=False, init_sh_input_img=True,
        feature_upsampler_channels=64, gaussian_regressor_channels=64, supervise_intermediate_depth=True,
        return_depth=True, train_depth_only=False, monodepth_vit_type="vits", local_mv_match=2,
    )  # fmt: skip
    enc = EncoderDepthSplat(cfg)
    torch.hub.load = load
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    enc.load_state_dict({k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}, strict=True)
    return enc.eval()


def shepperd(r: np.ndarray) -> np.ndarray:
    """Rotation matrices `[N, 3, 3]` -> unit quaternions `[N, 4]` `(w, x, y, z)`, `w >= 0` (Shepperd's method)."""
    out = []
    for m in r:
        t = np.trace(m)
        i = int(np.argmax([t, m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = 2 * np.sqrt(1 + t)
            q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        else:
            a, b, c = (i - 1, i % 3, (i + 1) % 3)
            s = 2 * np.sqrt(1 + m[a, a] - m[b, b] - m[c, c])
            q = np.zeros(4)
            q[0], q[1 + a], q[1 + b], q[1 + c] = (
                (m[c, b] - m[b, c]) / s,
                s / 4,
                (m[a, b] + m[b, a]) / s,
                (m[a, c] + m[c, a]) / s,
            )
        q = np.asarray(q, np.float64)
        out.append(q * (1 if q[0] >= 0 else -1) / np.linalg.norm(q))
    return np.stack(out)


def hamilton(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion products `a b` `(w, x, y, z)`."""
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def infer(
    enc: torch.nn.Module, img: torch.Tensor, cameras: tuple[torch.Tensor, torch.Tensor, float, float], long_side: int
) -> tuple[dict, dict, torch.Tensor, int, int]:
    """Upstream forward (b = 1) on `img` `[V, 3, H, W]` (0-1) with the runner's resizing and normalisation.

    `cameras`: pixel intrinsics, cam_to_world, near, far. Returns upstream's output (depths, means, covariances and
    `dump["scales"]` back in the input units), its `visualization_dump`, the confidence at the network size, and that
    size.
    """
    k, c2w, near, far = cameras
    s = 1.0 / near  # the runner's scene scale (near = 1)
    c2w = c2w.clone()
    c2w[:, :3, 3] *= s
    near, far = near * s, far * s
    v, _, h0, w0 = img.shape
    f = long_side / max(h0, w0)
    h, w = (max(16, round(x * f / 16) * 16) for x in (h0, w0))
    x = F.interpolate(img, (h, w), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    k = k.double().clone()
    k[:, :2, 2] += 0.5
    k[:, 0] /= w0
    k[:, 1] /= h0
    near_t, far_t = (torch.full((1, v), d, device=img.device) for d in (near, far))
    probs = []
    predictor = enc.depth_predictor.forward

    def hook(*a: object, **kw: object) -> dict:
        res = predictor(*a, **kw)
        probs.append(res["match_probs"][-1])
        return res

    enc.depth_predictor.forward = hook
    context = {"image": x[None], "intrinsics": k.float()[None], "extrinsics": c2w.float()[None], "near": near_t,
               "far": far_t}  # fmt: skip
    dump: dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            out = enc(context, global_step=0, visualization_dump=dump)
    finally:
        enc.depth_predictor.forward = predictor
    g = out["gaussians"]
    out["depths"] = out["depths"] / s
    g.means, g.covariances = g.means / s, g.covariances / s**2
    dump["scales"] = dump["scales"] / s
    prob = F.interpolate(probs[0].max(dim=1, keepdim=True)[0], size=(h, w), mode="nearest")
    return out, dump, prob, h, w


def predict(enc: torch.nn.Module, s: dict[str, np.ndarray], long_side: int) -> dict[str, np.ndarray]:
    """Upstream forward (b = 1) on the scene `s` with the runner's pre/post-processing."""
    img = torch.from_numpy(s["images"]).permute(0, 3, 1, 2).float() / 255
    h0, w0 = img.shape[2:]
    c2w = torch.from_numpy(s["cam_to_world"]).float()
    near, far = (float(s["depth_range"][i]) for i in (0, 1))
    out, dump, prob, h, w = infer(enc, img, (torch.from_numpy(s["intrinsics"]), c2w, near, far), long_side)
    g = out["gaussians"]
    depth = out["depths"][0]  # [V, h, w]

    def up(t: torch.Tensor) -> np.ndarray:
        return F.interpolate(t.float(), (h0, w0), mode="bilinear", align_corners=False)[:, 0].numpy()

    scales = dump["scales"][0].double().numpy()
    q_cam = dump["rotations"][0].double().numpy()[:, [3, 0, 1, 2]]
    q_cam /= np.linalg.norm(q_cam, axis=1, keepdims=True)
    q_c2w = np.repeat(shepperd(c2w[:, :3, :3].double().numpy()), h * w, 0)
    q = hamilton(q_c2w, q_cam)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    rot = torch.from_numpy(q[:, [1, 2, 3, 0]])  # back to upstream's x y z w for its own covariance builder
    from src.model.encoder.common.gaussians import build_covariance

    cov = build_covariance(torch.from_numpy(scales), rot).numpy()
    err = np.abs(cov - g.covariances[0].double().numpy()).max()
    print(f"  world rotations vs upstream covariance: max |diff| {err:.3g} (|cov| max {np.abs(cov).max():.3g})")
    if err >= 1e-5 * max(1.0, np.abs(cov).max()):
        msg = f"world rotations disagree with upstream's covariance: {err:.3g}"
        raise RuntimeError(msg)
    return {
        "depth": up(depth[:, None]),
        "confidence": up(prob),
        "gaussians.means": g.means[0].numpy(),
        "gaussians.scales": scales,
        "gaussians.rotations": q,
        "gaussians.harmonics": g.harmonics[0].numpy(),
        "gaussians.opacities": g.opacities[0].numpy(),
    }


def main() -> None:
    sys.path.append(sys.argv[4])
    enc = network(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    s = scene()
    maps = predict(enc, s, LONG_SIDE)
    h0, w0 = s["images"].shape[1:3]
    h, w = (max(16, round(x * LONG_SIDE / max(h0, w0) / 16) * 16) for x in (h0, w0))
    write(
        "depthsplat-small-re10k@1",
        {k: np.asarray(v, np.float32) for k, v in maps.items()},
        family="depthsplat",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, synth scene (4 views) at {h0}x{w0} -> network {h}x{w}, torch {torch.__version__}",
        task="multiview",
    )
    gt, d = s["depth"], maps["depth"]
    print(f"  depth AbsRel {np.mean(np.abs(d - gt) / gt):.4f}, {len(maps['gaussians.means'])} gaussians")


if __name__ == "__main__":
    main()
