"""ReSplat goldens from the reference implementation.

Reference: upstream cvg/resplat (`REF_COMMIT`), CPU float32, `EncoderReSplat.forward` then `forward_update` (eval, the
released configs: `config/model/encoder/resplat.yaml` with the overrides of `scripts/re10k_view2_256x256_base_refine.sh`
/ `scripts/dl3dv_view8_256x448_small_refine.sh`, `config/experiment/dl3dv.yaml`; 2 refinement steps) on the 4 views of
the `tests/synth.py` scene, with the runner's pre- and post-processing: images resized to long side `LONG_SIDE`
(bicubic antialiased, clamped), intrinsics normalised by the input size (pixel centres at `(i + 0.5) / W`), upstream's
fixed near / far, poses expressed in the frame of the middle view (upstream `pose_align_middle_view`), depth and
confidence (the largest match probability) resized back (bilinear).

torchvision's ResNet-18 is built without its ImageNet download (the checkpoint holds the weights). Two CUDA-only
pieces are replaced: `pointops` (kNN; brute force with exact squared distances) and the gsplat
rasteriser in the refinement loop (`fastforks.multiview.models.resplat.render`, the same `rasterization` call in plain
PyTorch, checked against gsplat on the GPU separately).

The Gaussians are mapped back to the input frame: centres exactly, rotations `q(R_c2w) q_cam` (`w x y z`, `w >= 0`,
Shepperd's method in float64; checked against upstream's covariance), SH coefficients by per-degree least-squares
rotation matrices of gsplat's basis.

    FASTFORKS_GOLDEN_PATH=... python -I tools/golden/resplat.py RESPLAT_CHECKOUT VARIANT CKPT EXTRA_SITE

`VARIANT`: `resplat-small-dl3dv` or `resplat-base-re10k`; `EXTRA_SITE`: a directory with `einops`, `jaxtyping` and
`e3nn` (upstream imports them), e.g. a `uv pip install --target`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "src")]  # python -I
# ruff: noqa: E402 - imports after the path setup

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write
from depthsplat import hamilton, shepperd

from synth import scene

UPSTREAM = "https://github.com/cvg/resplat"
REF_COMMIT = "cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1"
PACKAGES = (
    "src", "src.model", "src.model.encoder", "src.model.encoder.common", "src.model.encoder.unimatch",
    "src.model.encoder.point_transformer", "src.model.encoder.visualization", "src.model.decoder", "src.geometry",
    "src.misc", "src.dataset", "src.dataset.shims",
)  # fmt: skip
# variant: (config overrides, near, far, multiple, long side of the golden)
VARIANTS = {
    "resplat-small-dl3dv": (
        {"monodepth_vit_type": "vits", "gaussian_regressor_channels": 256, "shim_patch_size": 16, "upsample_factor": 8,
         "lowest_feature_resolution": 8},
        0.01, 200.0, 32, 256,
    ),
    "resplat-base-re10k": (
        {"latent_downsample": 2, "fixed_latent_size": False, "init_gaussian_multiple": 4,
         "refine_same_num_points": True},
        0.01, 100.0, 16, 128,
    ),
}  # fmt: skip
CONFIG = {  # config/model/encoder/resplat.yaml at REF_COMMIT, with num_refine = 2
    "num_depth_candidates": 128, "unimatch_weights_path": None, "downscale_factor": 4, "shim_patch_size": 4,
    "num_scales": 1, "upsample_factor": 4, "lowest_feature_resolution": 4, "depth_unet_channels": 128,
    "grid_sample_disable_cudnn": False, "local_mv_match": 2, "gaussian_regressor_channels": 512,
    "supervise_intermediate_depth": True, "return_depth": True, "sample_log_depth": True,
    "bilinear_upsample_depth": False, "no_upsample_depth": False, "monodepth_vit_type": "vitb",
    "attn_proj_channels": 64, "knn_samples": 16, "num_blocks": 6, "init_use_local_knn": False,
    "init_local_knn_spatial_radius": 3, "init_local_knn_num_neighbor_views": 32, "init_local_knn_cross_view_radius": 3,
    "latent_downsample": 4, "fixed_latent_size": True, "init_gaussian_multiple": 1, "refine_same_num_points": False,
    "depth_pred_half_res": False, "no_crop_image": False, "num_refine": 2, "train_min_refine": 0,
    "train_max_refine": 0, "num_basic_refine_blocks": 4, "state_channels": 512, "update_attn_proj_channels": 64,
    "refine_knn_samples": 8, "refine_use_local_knn": False, "refine_local_knn_spatial_radius": 3,
    "refine_local_knn_num_neighbor_views": 32, "refine_local_knn_cross_view_radius": 3,
    "render_error_mv_attn_blocks": 1, "use_amp": True, "pt_head_amp": True, "pt_update_amp": True,
    "use_checkpointing": False, "init_use_checkpointing": False, "recurrent_use_checkpointing": False,
}  # fmt: skip


def _pointops() -> types.ModuleType:
    """`pointops.knn_query` / `knn_query_and_group` for one batch: brute force, exact squared distances."""

    def knn_query(k: int, p: torch.Tensor, *_a: object) -> tuple[torch.Tensor, None]:
        rows = max(1, (1 << 26) // len(p))
        idx = [
            torch.cdist(p[i : i + rows], p, compute_mode="donot_use_mm_for_euclid_dist").topk(k, 1, largest=False)[1]
            for i in range(0, len(p), rows)
        ]
        return torch.cat(idx).int(), None

    def knn_query_and_group(feat: torch.Tensor, xyz: torch.Tensor, *_a: object, idx: torch.Tensor | None = None,
                            nsample: int = 16, with_xyz: bool = False,
                            **_kw: object) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
        if with_xyz:
            raise NotImplementedError
        if idx is None:
            idx = knn_query(nsample, xyz)[0]
        return feat[idx.long()], idx

    m = types.ModuleType("pointops")
    m.knn_query, m.knn_query_and_group = knn_query, knn_query_and_group
    return m


def network(repo: Path, variant: str, ckpt: Path) -> torch.nn.Module:
    """`EncoderReSplat` with the released config of `variant` from the checkout at `REF_COMMIT`, CPU float32, eval."""
    tree = reference_tree(repo, REF_COMMIT)
    for name in PACKAGES:  # empty parents: their __init__ pull in training code, datasets and gsplat
        m = types.ModuleType(name)
        m.__path__ = [str(tree / name.replace(".", "/"))]
        sys.modules[name] = m
    sys.modules["pointops"] = _pointops()
    import torchvision.models

    resnet18 = torchvision.models.resnet18  # upstream asks for ImageNet weights; the checkpoint holds them
    torchvision.models.resnet18 = lambda **_kw: resnet18(weights=None)
    # ruff: noqa: PLC0415 - reference checkout
    from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
    from src.model.encoder.encoder_resplat import EncoderReSplat, EncoderReSplatCfg
    from src.model.encoder.visualization.encoder_visualizer_resplat_cfg import EncoderVisualizerReSplatCfg

    adapter = GaussianAdapterCfg(
        gaussian_scale_min=0.5, gaussian_scale_max=1.0, sh_degree=3, exp_scale=False, softplus_scale=True,
        clamp_min_scale=1e-6, scale_detach_depth=False, exp_scale_bias=4.0, no_rotate_sh=True, no_sh_mask=True,
        init_rotation_identity=False,
    )  # fmt: skip
    vis = EncoderVisualizerReSplatCfg(num_samples=8, min_resolution=256, export_ply=False)
    cfg = EncoderReSplatCfg(name="resplat", visualizer=vis, gaussian_adapter=adapter, **(CONFIG | VARIANTS[variant][0]))
    enc = EncoderReSplat(cfg)
    torchvision.models.resnet18 = resnet18
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    enc.load_state_dict({k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}, strict=True)
    return enc.eval()


class Renderer:
    """Upstream's gsplat decoder call (`renderer.forward`) on the plain-PyTorch rasteriser."""

    def forward(  # noqa: PLR0913, PLR0917 - upstream's decoder signature
        self, g: object, c2w: torch.Tensor, k: torch.Tensor, near: torch.Tensor, far: torch.Tensor,
        size: tuple[int, int],
    ) -> SimpleNamespace:  # fmt: skip
        """Render the views of `g` (upstream `DecoderOutput` with `.color`)."""
        from fastforks.multiview.models.resplat.render import render

        img = render(g.means[0], g.covariances[0], g.opacities[0], g.harmonics[0], c2w[0], k[0], *size,
                     float(near[0, 0]), float(far[0, 0]))  # fmt: skip
        return SimpleNamespace(color=img[None])


def sh_basis(d: np.ndarray) -> np.ndarray:
    """The degree-3 SH basis of gsplat (`_eval_sh_bases_fast`) of unit directions `[N, 3]` -> `[N, 16]`."""
    x, y, z = d.T
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    return np.stack(
        (
            np.full_like(x, 0.28209479177387814),
            -0.48860251190291987 * y, 0.48860251190291987 * z, -0.48860251190291987 * x,
            1.0925484305920792 * xy, -1.0925484305920792 * yz, 0.94617469575755997 * zz - 0.31539156525251999,
            -1.0925484305920792 * xz, 0.54627421529603959 * (xx - yy),
            0.59004358992664352 * y * (-3 * xx + yy), 2.8906114426405538 * xy * z,
            0.45704579946446572 * y * (1 - 5 * zz), 0.3731763325901154 * z * (5 * zz - 3),
            0.45704579946446572 * x * (1 - 5 * zz), 1.4453057213202769 * z * (xx - yy),
            0.59004358992664352 * x * (-xx + 3 * yy),
        ),
        -1,
    )  # fmt: skip


def sh_rotation(r: np.ndarray) -> np.ndarray:
    """Per-degree matrices `D` (`[16, 16]`): coefficients `c @ D` at direction `R d` give the colour of `c` at `d`."""
    d = np.random.default_rng(1).standard_normal((200, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    a, b = sh_basis(d), sh_basis(d @ r.T)  # c B(d) = (c D) B(R d)  <=>  B(d) = D B(R d)  <=>  a = b D^T
    out = np.zeros((16, 16))
    for lo, hi in ((0, 1), (1, 4), (4, 9), (9, 16)):
        out[lo:hi, lo:hi] = np.linalg.lstsq(b[:, lo:hi], a[:, lo:hi], rcond=None)[0]
    if np.abs(a - b @ out).max() > 1e-5:  # noqa: PLR2004 - r is an fp32 rotation: orthogonal to ~1e-7
        msg = "SH rotation fit failed"
        raise RuntimeError(msg)
    return out.T


def predict(enc: torch.nn.Module, s: dict[str, np.ndarray], variant: str) -> dict[str, np.ndarray]:
    """Upstream forward + refinement (b = 1) on the scene `s` with the runner's pre/post-processing."""
    _, near, far, multiple, long_side = VARIANTS[variant]
    img = torch.from_numpy(s["images"]).permute(0, 3, 1, 2).float() / 255
    v, _, h0, w0 = img.shape
    f = long_side / max(h0, w0)
    h, w = (max(multiple, round(x * f / multiple) * multiple) for x in (h0, w0))
    x = F.interpolate(img, (h, w), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    k = torch.from_numpy(s["intrinsics"]).double().clone()
    k[:, :2, 2] += 0.5
    k[:, 0] /= w0
    k[:, 1] /= h0
    c2w = torch.from_numpy(s["cam_to_world"]).double()
    mid = c2w[v // 2]
    c2w_net = torch.linalg.inv(mid)[None] @ c2w
    near_t, far_t = (torch.full((1, v), d) for d in (near, far))
    context = {"image": x[None], "intrinsics": k.float()[None], "extrinsics": c2w_net.float()[None], "near": near_t,
               "far": far_t}  # fmt: skip
    probs = []
    predictor = enc.depth_predictor.forward

    def hook(*a: object, **kw: object) -> dict:
        res = predictor(*a, **kw)
        probs.append(res["match_probs"][-1])
        return res

    enc.depth_predictor.forward = hook
    try:
        with torch.no_grad():
            out = enc(context, global_step=0)
            ref = enc.forward_update(context, None, out["condition_features"], out["gaussians"], Renderer())
    finally:
        enc.depth_predictor.forward = predictor
    g = ref["gaussian"][-1]

    def up(t: torch.Tensor) -> np.ndarray:
        return F.interpolate(t.float(), (h0, w0), mode="bilinear", align_corners=False)[:, 0].numpy()

    rm, tm = mid[:3, :3].numpy(), mid[:3, 3].numpy()
    means = g.means[0].double().numpy() @ rm.T + tm
    q_cam = g.rotations[0].double().numpy()[:, [3, 0, 1, 2]]
    q_cam /= np.linalg.norm(q_cam, axis=1, keepdims=True)
    q = hamilton(np.repeat(shepperd(c2w[:, :3, :3].numpy()), len(q_cam) // v, 0), q_cam)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    from src.model.encoder.common.gaussians import build_covariance

    scales = g.scales[0].double()
    cov = build_covariance(scales, torch.from_numpy(q[:, [1, 2, 3, 0]])).numpy()
    want = rm @ g.covariances[0].double().numpy() @ rm.T
    err = np.abs(cov - want).max()
    print(f"  world rotations vs upstream covariance: max |diff| {err:.3g} (|cov| max {np.abs(want).max():.3g})")
    if err >= 1e-5 * max(1.0, np.abs(want).max()):
        msg = f"world rotations disagree with upstream's covariance: {err:.3g}"
        raise RuntimeError(msg)
    return {
        "depth": up(out["depths"][0][:, None]),
        "confidence": up(probs[0].max(dim=1, keepdim=True)[0]),
        "gaussians.means": means,
        "gaussians.scales": scales.numpy(),
        "gaussians.rotations": q,
        "gaussians.harmonics": g.harmonics[0].double().numpy() @ sh_rotation(rm),
        "gaussians.opacities": g.opacities[0].numpy(),
    }


def main() -> None:
    sys.path.append(sys.argv[4])
    variant = sys.argv[2]
    enc = network(Path(sys.argv[1]), variant, Path(sys.argv[3]))
    s = scene()
    maps = predict(enc, s, variant)
    _, _, _, multiple, long_side = VARIANTS[variant]
    h0, w0 = s["images"].shape[1:3]
    h, w = (max(multiple, round(x * long_side / max(h0, w0) / multiple) * multiple) for x in (h0, w0))
    write(
        f"{variant}@1",
        {k: np.asarray(v, np.float32) for k, v in maps.items()},
        family="resplat",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, synth scene (4 views) at {h0}x{w0} -> network {h}x{w}, 2 refinement steps, torch "
        f"{torch.__version__}",
        task="multiview",
    )
    gt, d = s["depth"], maps["depth"]
    print(f"  depth AbsRel {np.mean(np.abs(d - gt) / gt):.4f}, {len(maps['gaussians.means'])} gaussians")


if __name__ == "__main__":
    main()
